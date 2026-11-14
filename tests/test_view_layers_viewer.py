"""The View > Layers submenu, driven through a real offscreen ViewerWindow.

Same technique and default-tier rationale as
tests/test_sprite_toggle_viewer.py, which this module is modelled on: assert
on QAction handles and on the LIVE chunk cache's own state, never on widget
isVisible() (unconditionally False for a never-shown window under
QT_QPA_PLATFORM=offscreen).

Two of these pin design decisions rather than behaviour, and both are the
kind a later refactor "tidies" into a bug:

- greying an action must never mutate the window's LayerState, or the toggle
  silently flips itself on a style round trip;
- a toggle must apply ONCE, not once per layer or once per style -- this
  feature adds a wholesale-clear caller to a render path that is actively
  trying to shed them.

Every ViewerWindow() constructed here goes through conftest.close_window() --
see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from descape import asset_source, settings, unit_sprites, view_layers, viewer_unit_place
from descape.settings_dialog import SettingsDialog
from testkit.viewer_family import set_viewer_global

import conftest
from test_unit_sprites import CONST, FILE_NAME, build_sld

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FARM_LAYER = "farm_overlay"
TEXTURES_LAYER = "terrain_textures"
SMALL_TREES_LAYER = "small_trees"


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    """A tmp install holding one small synthetic sprite, so Show sprites is
    reachable at all -- tests/test_sprite_toggle_viewer.py's fixture of the
    same name, for the same reason (the farm layer is requires_sprites, so
    without this it is permanently greyed and half this file is vacuous)."""
    # A hover can cache the fake table's empty wall set for later tests in the same worker.
    viewer_unit_place._wall_consts.cache_clear()
    assert viewer_unit_place._wall_consts(), "cache the real wall family before patching its source"
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                         "mirroring_mode": 6, "frame_count": 1}},
    )
    asset_source.set_install_path_override(tmp_path)
    yield tmp_path
    asset_source.set_install_path_override(None)


def test_the_synthetic_sprite_install_keeps_the_real_wall_run_consts(sprite_install) -> None:
    assert {72, 117, 231} <= viewer_unit_place._wall_consts(), "the fake sprite table poisoned the cached wall family"


def test_the_submenu_has_one_checked_action_per_registry_row() -> None:
    window = conftest.blank_window()
    try:
        assert set(window.layer_actions) == {spec.layer_id for spec in view_layers.LAYERS}
        for spec in view_layers.LAYERS:
            action = window.layer_actions[spec.layer_id]
            assert action.isCheckable(), spec.layer_id
            assert action.isChecked() is spec.default, spec.layer_id
    finally:
        conftest.close_window(window)


def test_it_lives_under_view_not_in_filters() -> None:
    """Same placement rule View > Show sprites is pinned to: Filters is
    WHICH units, View is everything else about the render -- including how
    big one category of unit art is drawn."""
    window = conftest.blank_window()
    try:
        view_menu = next(
            action.menu() for action in window.menuBar().actions()
            if action.text().replace("&", "") == "View"
        )
        submenu = next(
            action.menu() for action in view_menu.actions()
            if action.menu() is not None and action.text().replace("&", "") == "Layers"
        )
        labels = {action.text() for action in submenu.actions()}
        assert labels == {spec.label for spec in view_layers.LAYERS}
    finally:
        conftest.close_window(window)


def test_every_layer_is_greyed_with_no_map() -> None:
    window = conftest.blank_window(load=False)
    try:
        for layer_id, action in window.layer_actions.items():
            assert not action.isEnabled(), layer_id
            assert "Open a map" in action.toolTip(), layer_id
    finally:
        conftest.close_window(window)


def test_gating_matches_the_registry_in_every_style(sprite_install) -> None:
    """Keyed on window._render_style, not the combo: Flat + Isometric View
    (the default) renders through IsoChunkCache, where farms really do
    drape, so gating on the terrain style would grey a live layer."""
    window = conftest.blank_window()
    try:
        for style in ("Stepped", "Flat", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            for spec in view_layers.LAYERS:
                expected, tip = view_layers.availability(
                    spec, style=window._render_style,
                    sprites_enabled=window._sprites_enabled, has_map=True, has_mod_art=False,
                )
                action = window.layer_actions[spec.layer_id]
                assert action.isEnabled() is expected, (style, spec.layer_id)
                assert action.toolTip() == tip, (style, spec.layer_id)
    finally:
        conftest.close_window(window)


def test_the_farm_layer_greys_the_moment_sprites_go_off(sprite_install) -> None:
    """requires_sprites has to be re-gated from _on_sprites_toggled itself:
    nothing else fires between the sprite flip and the user looking at the
    menu."""
    window = conftest.blank_window()
    try:
        window.show_sprites_action.setChecked(True)
        assert window.layer_actions[FARM_LAYER].isEnabled()

        window.show_sprites_action.setChecked(False)
        action = window.layer_actions[FARM_LAYER]
        assert not action.isEnabled()
        assert "Show sprites" in action.toolTip()
    finally:
        conftest.close_window(window)


def test_layers_persist_across_windows(sprite_install, monkeypatch) -> None:
    """GH #182 reversed GH #85's per-session rule: a fresh window starts with
    the rows as the last window left them, cache included. The memo is
    nulled so the second window reads config.yaml, not the in-process copy."""
    first = conftest.blank_window()
    try:
        for layer_id in (TEXTURES_LAYER, FARM_LAYER):
            first.layer_actions[layer_id].setChecked(False)
        first.layer_actions[SMALL_TREES_LAYER].setChecked(True)
        assert first._cache.layers.terrain_textures is False
    finally:
        conftest.close_window(first)

    monkeypatch.setattr(settings, "_view_layers", None)
    second = conftest.blank_window()
    try:
        assert second.layer_actions[TEXTURES_LAYER].isChecked() is False
        assert second.layer_actions[FARM_LAYER].isChecked() is False
        assert second.layer_actions[SMALL_TREES_LAYER].isChecked() is True
        assert second._layers == view_layers.LayerState(
            terrain_textures=False, farm_overlay=False, small_trees=True
        )
        assert second._cache.layers.terrain_textures is False
        assert second._cache.layers.farm_overlay is False
    finally:
        conftest.close_window(second)


def test_the_farm_row_stays_unticked_across_a_sprites_off_and_on(sprite_install) -> None:
    """The sprite round trip greys the farm row and brings it back, which
    must not reset the user's untick."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Stepped")
        window.show_sprites_action.setChecked(True)
        window.layer_actions[FARM_LAYER].setChecked(False)

        window.show_sprites_action.setChecked(False)
        assert not window.layer_actions[FARM_LAYER].isEnabled()
        window.show_sprites_action.setChecked(True)

        action = window.layer_actions[FARM_LAYER]
        assert action.isEnabled()
        assert action.isChecked() is False
        assert window._layers.farm_overlay is False
        assert window._cache.layers.farm_overlay is False
    finally:
        conftest.close_window(window)


def test_toggling_reaches_the_live_cache(sprite_install) -> None:
    window = conftest.blank_window()
    try:
        window.layer_actions[TEXTURES_LAYER].setChecked(False)
        assert window._layers.terrain_textures is False
        assert window._cache.layers.terrain_textures is False

        window.layer_actions[TEXTURES_LAYER].setChecked(True)
        assert window._cache.layers.terrain_textures is True
    finally:
        conftest.close_window(window)


def test_state_survives_a_style_switch(sprite_install) -> None:
    """_render_current() builds a brand-new cache per style, so the window's
    own copy is the only thing that remembers the choice."""
    window = conftest.blank_window()
    try:
        window.layer_actions[TEXTURES_LAYER].setChecked(False)
        for style in ("Flat", "Sloped", "Stepped"):
            window.terrain_style_combo.setCurrentText(style)
            assert window._layers.terrain_textures is False, style
            assert window._cache.layers.terrain_textures is False, style
    finally:
        conftest.close_window(window)


def test_state_survives_a_reopen(sprite_install) -> None:
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.blank_window()
    try:
        window.layer_actions[TEXTURES_LAYER].setChecked(False)
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window._cache.layers.terrain_textures is False
    finally:
        conftest.close_window(window)


def test_greying_never_mutates_the_state(sprite_install) -> None:
    """The decision this pins: an unavailable layer keeps its checked state
    and re-applies when it becomes available again. Resetting it on grey
    looks tidy and silently flips the toggle on a style round trip.

    Isometric View has to come OFF for the Flat leg: left on (the default),
    Flat renders through IsoChunkCache and Small Trees stays live, so the
    greying half of this test would never fire. Small Trees, not the farm
    row, which top-down Flat drapes too since GH #150."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Stepped")
        window.layer_actions[SMALL_TREES_LAYER].setChecked(True)
        assert window._layers.small_trees is True

        window.terrain_style_combo.setCurrentText("Flat")
        window.iso_action.setChecked(False)
        assert window._render_style == "flat", "fixture no longer reaches real Flat"
        assert not window.layer_actions[SMALL_TREES_LAYER].isEnabled()
        assert window.layer_actions[SMALL_TREES_LAYER].isChecked() is True
        assert window._layers.small_trees is True

        window.terrain_style_combo.setCurrentText("Stepped")
        assert window.layer_actions[SMALL_TREES_LAYER].isEnabled()
        assert window.layer_actions[SMALL_TREES_LAYER].isChecked() is True
        assert window._cache.layers.small_trees is True
    finally:
        conftest.close_window(window)


def test_a_toggle_applies_once_not_n_times(sprite_install, monkeypatch) -> None:
    """A wholesale canvas clear is the most expensive thing this path can
    do, and the draw-perf work is actively reducing the number of callers
    that make one. One toggle must cost exactly one."""
    window = conftest.blank_window()
    try:
        calls = []
        real = type(window._cache).set_layers
        monkeypatch.setattr(
            type(window._cache), "set_layers",
            lambda self, layers: (calls.append(layers), real(self, layers))[1],
        )
        window.layer_actions[TEXTURES_LAYER].setChecked(False)
        assert len(calls) == 1, calls
    finally:
        conftest.close_window(window)


def test_toggling_with_no_map_open_is_remembered_not_dropped() -> None:
    """The window's own copy is stored unconditionally, before the
    no-map early return, so a choice made on an empty window reaches the
    next document."""
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.blank_window(load=False)
    try:
        window.layer_actions[TEXTURES_LAYER].setChecked(False)
        assert window._layers.terrain_textures is False
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window._cache.layers.terrain_textures is False
    finally:
        conftest.close_window(window)


FISH_LAYER = "fish_outlines"


def test_the_fish_toggle_reaches_map_view_never_the_cache_and_persists(monkeypatch) -> None:
    """TASK-499: a scene-overlay row goes to MapView alone. No set_layers()
    call, no cache rebuild, and the pick index it draws from is built outside
    Units mode; the choice survives a restart like every other row."""
    window = conftest.blank_window()
    try:
        calls = []
        real = type(window._cache).set_layers
        monkeypatch.setattr(
            type(window._cache), "set_layers",
            lambda self, layers: (calls.append(layers), real(self, layers))[1],
        )
        cache = window._cache
        assert window.mode != "units" and window.map_view._unit_index is None
        action = window.layer_actions[FISH_LAYER]
        spec = next(s for s in view_layers.LAYERS if s.layer_id == FISH_LAYER)
        assert action.isEnabled() and action.toolTip() == spec.tooltip
        assert action.text() == spec.label and action.isChecked() is False
        action.setChecked(True)
        assert calls == [] and window._cache is cache
        assert window._layers.fish_outlines is True and settings.get_view_layers().fish_outlines is True
        assert cache.layers.fish_outlines is False
        view = window.map_view
        assert view._fish_enabled is True and view._fish_item.isVisible()
        assert view._unit_index is not None, "no index outside Units mode, so no ring would draw"
        assert window._needs_unit_index()
        assert window.status_log.toPlainText().splitlines()[-1] == "Fish Outlines: on"

        action.setChecked(False)
        assert calls == [] and window._cache is cache
        assert view._fish_enabled is False and not view._fish_item.isVisible()
        action.setChecked(True)
    finally:
        conftest.close_window(window)

    monkeypatch.setattr(settings, "_view_layers", None)
    second = conftest.blank_window()
    try:
        assert second.layer_actions[FISH_LAYER].isChecked() is True
        assert second._layers.fish_outlines is True
        assert second.map_view._fish_enabled is True and second.map_view._fish_item.isVisible()
        assert second.map_view._unit_index is not None
    finally:
        conftest.close_window(second)


def test_the_fish_row_survives_a_style_switch_and_another_layer_toggle(sprite_install) -> None:
    window = conftest.blank_window()
    try:
        window.layer_actions[FISH_LAYER].setChecked(True)
        for style in ("Flat", "Sloped", "Stepped"):
            window.terrain_style_combo.setCurrentText(style)
            assert window.map_view._fish_item.isVisible(), style
            assert window.map_view._unit_index is not None, style
        window.layer_actions[TEXTURES_LAYER].setChecked(False)
        assert window._cache.layers == view_layers.LayerState(terrain_textures=False)
        assert window.map_view._fish_enabled is True
    finally:
        conftest.close_window(window)


def test_the_small_trees_toggle_reaches_the_live_cache(sprite_install) -> None:
    """The size row, mirroring test_toggling_reaches_the_live_cache above:
    it is baked into the cache's sprites, so the window's copy alone reaching
    the new value would leave the map unchanged."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        window.layer_actions[SMALL_TREES_LAYER].setChecked(True)
        assert window._layers.small_trees is True
        assert window._cache.layers.small_trees is True
        assert window._cache.layers.tree_scale == view_layers.SMALL_TREE_SCALE

        window.layer_actions[SMALL_TREES_LAYER].setChecked(False)
        assert window._cache.layers.small_trees is False
        assert window._cache.layers.tree_scale == 1.0
    finally:
        conftest.close_window(window)


def test_the_small_trees_state_survives_a_style_switch(sprite_install) -> None:
    """_render_current() builds a brand-new cache per style. Checked on the
    default-OFF row too, not just a default-on one: an "is it still default"
    style of restore would pass for terrain_textures and fail here."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        window.layer_actions[SMALL_TREES_LAYER].setChecked(True)
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            assert window._layers.small_trees is True, style
            assert window._cache.layers.small_trees is True, style
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_pick_drapes_farms_only_under_the_renders_own_three_gates(sprite_install, style) -> None:
    """TASK-015: MapView._farms_draped() is the live cache's units shown,
    sprites on and Farm Overlay on, in Stepped as well as Sloped."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText(style)
        mv = window.map_view
        assert mv._canvas_item._cache is window._cache
        window.show_sprites_action.setChecked(True)
        window.layer_actions[FARM_LAYER].setChecked(True)
        assert window._cache.with_units and mv._farms_draped() is True

        window.layer_actions[FARM_LAYER].setChecked(False)
        assert mv._farms_draped() is False
        window.layer_actions[FARM_LAYER].setChecked(True)
        window.show_sprites_action.setChecked(False)
        assert mv._farms_draped() is False
    finally:
        conftest.close_window(window)


def test_pick_drapes_farms_in_top_down_flat_under_the_same_gates(sprite_install) -> None:
    """GH #150: top-down Flat drapes too, so pick follows the same gates."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Flat")
        window.show_sprites_action.setChecked(True)
        window.layer_actions[FARM_LAYER].setChecked(True)
        assert window._render_style == "stepped"  # Flat + Isometric View renders through IsoChunkCache
        assert window.map_view._farms_draped() is True
        window.iso_action.setChecked(False)
        assert window._render_style == "flat"
        assert window.map_view._canvas_item._cache is window._cache
        assert window.map_view._farms_draped() is True
        window.layer_actions[FARM_LAYER].setChecked(False)
        assert window.map_view._farms_draped() is False
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("toggle", ["farm_overlay", "sprites"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_farms_selection_and_hover_follow_a_drape_toggle(sprite_install, style, toggle) -> None:
    """TASK-015: a draped farm outlines per tile at each tile's own height,
    so Farm Overlay and Show sprites change its shape on uneven ground. The
    selection must redraw on the toggle, and the hover must not keep its
    old path through its same-unit memo on the next move."""
    from PyQt5.QtGui import QPainterPath

    window = conftest.blank_window()
    try:
        mm = window.scenario.map_manager
        for tile in mm.terrain:
            tile.elevation = (tile.x // 2 + tile.y // 3) % 3
        window._ensure_unit_edits().add(1, 50, 60.5, 60.5)
        window.terrain_style_combo.setCurrentText(style)
        window.refresh_map()  # the blank template already opens in Stepped
        window.show_sprites_action.setChecked(True)
        window.layer_actions[FARM_LAYER].setChecked(True)
        window._rebuild_unit_index()
        mv = window.map_view
        entry = next(e for e in mv._unit_index.entries if e.unit.unit_const == 50)
        assert mv._farms_draped() is True

        mv.set_unit_selection([entry])
        mv._update_unit_hover(entry)
        (outline,) = [group[2] for group in mv._unit_select_groups.values()]
        draped = QPainterPath(outline.path())
        assert mv._unit_hover_item.path() == draped

        if toggle == "farm_overlay":
            window.layer_actions[FARM_LAYER].setChecked(False)
        else:
            window.show_sprites_action.setChecked(False)
        assert mv._farms_draped() is False
        plain = mv._unit_path(entry)
        assert plain != draped, "the farm must straddle levels, or the toggle changes nothing to see"
        (outline,) = [group[2] for group in mv._unit_select_groups.values()]
        assert outline.path() == plain, "the selection kept the draped outline"
        mv._update_unit_hover(entry)  # the cursor's next move, still on the farm
        assert mv._unit_hover_item.path() == plain, "the hover kept the draped outline"
    finally:
        conftest.close_window(window)


def test_the_drag_ghost_takes_the_windows_own_layer_value(sprite_install, monkeypatch) -> None:
    """The ghost resolves its sprite outside every chunk cache, so it is the
    one path a default parameter would leave full-size -- a tree would drag
    big and snap small on drop. Stubbed at viewer's own imported name, since
    what is being pinned is which value the window hands over, not what the
    resolver does with it (tests/test_view_layers.py covers that)."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        window.show_sprites_action.setChecked(True)
        seen: list[float] = []
        set_viewer_global(
            monkeypatch, "unit_sprite_draws_at",
            lambda *args, **kwargs: (seen.append(kwargs["tree_scale"]), [])[1],
        )
        monkeypatch.setattr(window, "_ghost_rotation_override", lambda entry: None)
        ghost = SimpleNamespace(x=5.5, y=5.5, unit_const=CONST, rotation=0.0, reference_id=1)
        entry = SimpleNamespace(player_id=1)

        window.layer_actions[SMALL_TREES_LAYER].setChecked(True)
        assert window._drag_ghost_sprite_draws(entry, ghost) == []
        assert seen == [view_layers.SMALL_TREE_SCALE]

        window.layer_actions[SMALL_TREES_LAYER].setChecked(False)
        window._drag_ghost_sprite_draws(entry, ghost)
        assert seen[-1] == 1.0
    finally:
        conftest.close_window(window)


TRANSLUCENT_WALLS_LAYER = "translucent_walls"


def test_the_translucent_walls_toggle_reaches_the_live_cache_and_survives_a_style_switch(sprite_install) -> None:
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        window.layer_actions[TRANSLUCENT_WALLS_LAYER].setChecked(True)
        assert window._cache.layers.translucent_walls is True
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            assert window._layers.translucent_walls is True, style
            assert window._cache.layers.translucent_walls is True, style
        window.layer_actions[TRANSLUCENT_WALLS_LAYER].setChecked(False)
        assert window._cache.layers.wall_alpha == 1.0
    finally:
        conftest.close_window(window)


def test_the_drag_ghost_takes_the_windows_own_wall_alpha(sprite_install, monkeypatch) -> None:
    """test_the_drag_ghost_takes_the_windows_own_layer_value's shape: without
    it a dragged wall or gate shows opaque."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        window.show_sprites_action.setChecked(True)
        seen: list[float] = []
        set_viewer_global(
            monkeypatch, "unit_sprite_draws_at",
            lambda *args, **kwargs: (seen.append(kwargs["wall_alpha"]), [])[1],
        )
        monkeypatch.setattr(window, "_ghost_rotation_override", lambda entry: None)
        ghost = SimpleNamespace(x=5.5, y=5.5, unit_const=CONST, rotation=0.0, reference_id=1)
        entry = SimpleNamespace(player_id=1)

        window.layer_actions[TRANSLUCENT_WALLS_LAYER].setChecked(True)
        window._drag_ghost_sprite_draws(entry, ghost)
        assert seen[-1] == window._layers.wall_opacity_pct / 100

        window.layer_actions[TRANSLUCENT_WALLS_LAYER].setChecked(False)
        window._drag_ghost_sprite_draws(entry, ghost)
        assert seen[-1] == 1.0
    finally:
        conftest.close_window(window)


def test_the_opacity_setting_reaches_the_cache_only_while_translucent_walls_is_on(
    sprite_install, monkeypatch
) -> None:
    """With the row off the drawn alpha is unchanged, so set_layers() is not
    called at all: its equality guard would miss on the raw pct and evict the
    whole canvas for nothing."""
    window = conftest.blank_window()
    dialog = SettingsDialog(window)
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        calls = []
        real = type(window._cache).set_layers
        monkeypatch.setattr(
            type(window._cache), "set_layers",
            lambda self, layers: (calls.append(layers), real(self, layers))[1],
        )
        dialog.wall_opacity_slider.setValue(30)
        dialog._apply_wall_opacity()
        assert settings.get_wall_opacity_pct() == 30
        assert window._layers.wall_opacity_pct == 30
        assert calls == []

        window.layer_actions[TRANSLUCENT_WALLS_LAYER].setChecked(True)
        assert window._cache.layers.wall_alpha == 0.3
        dialog.wall_opacity_slider.setValue(70)
        dialog._apply_wall_opacity()
        assert len(calls) == 2
        assert window._cache.layers.wall_alpha == 0.7
    finally:
        dialog.deleteLater()
        conftest.close_window(window)


def test_the_opacity_slider_keeps_focus_through_an_applied_step(sprite_install) -> None:
    """_apply_layers() disables the window, and the modal dialog with it, so
    a keyboard user's next arrow press would go nowhere."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.blank_window()
    dialog = SettingsDialog(window)
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        window.layer_actions[TRANSLUCENT_WALLS_LAYER].setChecked(True)
        dialog.show()
        dialog.activateWindow()
        QApplication.setActiveWindow(dialog)
        QApplication.processEvents()
        dialog.wall_opacity_slider.setFocus()
        QApplication.processEvents()
        assert dialog.wall_opacity_slider.hasFocus()
        dialog.wall_opacity_slider.setValue(40)
        dialog._apply_wall_opacity()
        QApplication.processEvents()
        assert window._cache.layers.wall_alpha == 0.4
        assert dialog.wall_opacity_slider.hasFocus()
    finally:
        dialog.deleteLater()
        conftest.close_window(window)


def test_the_window_starts_with_the_saved_wall_opacity(sprite_install) -> None:
    settings.set_wall_opacity_pct(25)
    window = conftest.blank_window()
    try:
        assert window._layers.wall_opacity_pct == 25
        assert window._cache.layers.wall_opacity_pct == 25
    finally:
        conftest.close_window(window)


MOD_ART_LAYER = "mod_art"
MOD_STATUS_PATH = "subscribed//1_Green"


@pytest.fixture
def mod_art_profile(sprite_install, monkeypatch):
    """sprite_install plus a profile whose one mod replaces FILE_NAME with a
    1-frame .smx in plain green (no playercolor, so no tint can fake it)."""
    from test_smx_decoder import DRAW, _file, _frame

    palettes = sprite_install / "resources" / "_common" / "palettes"
    palettes.mkdir(parents=True)
    rows = "\n".join("0 250 0" for _ in range(1024))
    (palettes / "b_scen.pal").write_text(f"JASC-PAL\n0100\n1024\n{rows}\n")
    (palettes / "palettes.conf").write_text("28,b_scen.pal\n")
    profile = sprite_install / "profile"
    graphics = profile / "7656" / "mods" / "subscribed" / "1_Green" / "resources" / "_common" / "drs" / "graphics"
    graphics.mkdir(parents=True)
    rows = [(0, [(DRAW, [(1, 0)] * 48)])] * 24
    (graphics / f"{FILE_NAME}.smx").write_bytes(_file(_frame(rows, width=48, hotspot=(24, 12))))
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    return MOD_STATUS_PATH


def _unit_draws(window) -> list:
    from descape.render_cache import FlatChunkCache

    if isinstance(window._cache, FlatChunkCache):
        return list(window._cache._level_icons(0).values())
    return [draw for draws in window._cache._level(0).sprites.by_anchor.values() for draw, *_ in draws]


def _draws_mod_green(window) -> bool:
    draws = _unit_draws(window)
    assert draws, "the unit must draw something either way"
    return any(
        ((d.rgba[..., 1] > 200) & (d.rgba[..., 0] < 60) & (d.rgba[..., 2] < 60) & (d.rgba[..., 3] > 0)).any()
        for d in draws
    )


def _mod_art_window(style: str):
    from test_sprite_edit_bbox import Unit

    window = conftest.blank_window()
    # Player 2: the install sprite's full playercolor mask tints it red, never green.
    window.scenario.unit_manager.units[2].append(Unit(20.5, 20.5, CONST))
    window.terrain_style_combo.setCurrentText(style)
    if style == "Flat":
        window.iso_action.setChecked(False)
    window.show_sprites_action.setChecked(True)
    window.refresh_map()
    return window


@pytest.mark.parametrize("style", ["Stepped", "Flat"])
def test_the_mod_art_toggle_and_an_opt_out_redraw_from_the_files_that_win(mod_art_profile, style) -> None:
    """TASK-378: the toggle and Settings' per-mod tick both go through
    _apply_mod_art_change(), which must rebuild Stepped's sprites and Flat's
    icons from the new art, not recomposite the old."""
    from PyQt5.QtCore import Qt

    window = _mod_art_window(style)
    dialog = None
    try:
        action = window.layer_actions[MOD_ART_LAYER]
        assert action.isEnabled() and action.isChecked()
        assert window._mod_art_status() == "Mod art: 1 mod"
        assert _draws_mod_green(window)
        action.setChecked(False)
        assert not _draws_mod_green(window)
        assert window._mod_art_status() == ""
        assert settings.get_view_layers().mod_art is False
        action.setChecked(True)
        assert _draws_mod_green(window)

        dialog = SettingsDialog(window)
        item = dialog.mod_art_list.item(0)
        assert item.data(Qt.UserRole) == mod_art_profile and item.checkState() == Qt.Checked
        item.setCheckState(Qt.Unchecked)
        assert settings.get_mod_art_excluded() == {mod_art_profile}
        assert not _draws_mod_green(window)
        assert action.isChecked() and action.isEnabled(), "an opt-out leaves the row alone"
        item.setCheckState(Qt.Checked)
        assert _draws_mod_green(window)
    finally:
        if dialog is not None:
            dialog.deleteLater()
        conftest.close_window(window)


def test_a_window_opened_with_mod_art_persisted_off_draws_install_art(mod_art_profile) -> None:
    from descape import mod_graphics

    settings.set_view_layer(MOD_ART_LAYER, False)
    window = _mod_art_window("Stepped")
    try:
        assert not window.layer_actions[MOD_ART_LAYER].isChecked()
        assert not mod_graphics.enabled()
        assert not _draws_mod_green(window)
    finally:
        conftest.close_window(window)


def test_mod_art_is_greyed_when_no_mod_ships_art(sprite_install) -> None:
    window = conftest.blank_window()
    try:
        action = window.layer_actions[MOD_ART_LAYER]
        assert not action.isEnabled() and action.toolTip().startswith("No mod in the profile ships art")
        assert action.isChecked(), "greying never unchecks"
    finally:
        conftest.close_window(window)


def test_a_window_seeds_mod_art_opt_outs_and_textures_before_any_panel(mod_art_profile) -> None:
    """The seed runs before TerrainPanel, whose swatches resolve (and cache)
    texture paths at construction; a persisted opt-out reaches the first render."""
    from PIL import Image

    from descape import mod_graphics

    textures = sprite_install_profile_textures(mod_art_profile)
    names = set(asset_source._terrain_texture_map().values())
    for name in names:
        Image.new("RGB", (4, 4), (0, 250, 0)).save(textures / name, format="DDS")
    window = _mod_art_window("Stepped")
    try:
        for tid, name in asset_source._terrain_texture_map().items():
            assert asset_source.get_terrain_texture_path(tid) == textures / name, tid
    finally:
        conftest.close_window(window)

    settings.set_mod_art_excluded({mod_art_profile})
    window = _mod_art_window("Stepped")
    try:
        assert mod_graphics.enabled() and window.layer_actions[MOD_ART_LAYER].isChecked()
        assert not _draws_mod_green(window)
    finally:
        conftest.close_window(window)


def sprite_install_profile_textures(status_path: str):
    import os
    from pathlib import Path

    folder = status_path.split("//")[1]
    root = Path(os.environ["AOE2DE_PROFILE_PATH"]) / "7656" / "mods" / "subscribed" / folder / "resources" / "_common"
    textures = root / "terrain" / "textures" / "2x"
    textures.mkdir(parents=True, exist_ok=True)
    return textures


@pytest.fixture
def disabled_mod_art_profile(mod_art_profile):
    """mod_art_profile with its one mod disabled in the game's mod-status.json (GH #264)."""
    import json
    import os
    from pathlib import Path

    status = {"Mods": [{"Enabled": False, "Path": mod_art_profile, "Priority": 1}]}
    (Path(os.environ["AOE2DE_PROFILE_PATH"]) / "7656" / "mods" / "mod-status.json").write_text(json.dumps(status))
    return mod_art_profile


def test_a_mod_disabled_in_the_game_lists_off_and_draws_once_ticked(disabled_mod_art_profile) -> None:
    """GH #264: Settings > Mods lists a mod the game disables, marked and
    unticked, and the Mod Art row stays live for it. Its tick writes
    mod_art_included (never the opt-outs), draws its art and logs the plain
    title; a reopened dialog shows it ticked; unticking reverts."""
    from PyQt5.QtCore import Qt

    window = _mod_art_window("Stepped")
    dialog = reopened = None
    try:
        action = window.layer_actions[MOD_ART_LAYER]
        assert action.isEnabled() and action.isChecked()
        assert not _draws_mod_green(window)
        assert window._mod_art_status() == ""

        dialog = SettingsDialog(window)
        item = dialog.mod_art_list.item(0)
        assert item.data(Qt.UserRole) == disabled_mod_art_profile
        assert item.text() == "1_Green (disabled in game)" and item.checkState() == Qt.Unchecked
        assert item.toolTip().splitlines()[0] == "Disabled in the game's mod list, drawn here only when ticked"
        item.setCheckState(Qt.Checked)
        assert settings.get_mod_art_included() == {disabled_mod_art_profile}
        assert settings.get_mod_art_excluded() == frozenset()
        assert _draws_mod_green(window)
        assert window._mod_art_status() == "Mod art: 1 mod"
        log = window.status_log.toPlainText()
        assert "Mod art from 1_Green: on" in log and "(disabled in game)" not in log

        reopened = SettingsDialog(window)
        assert reopened.mod_art_list.item(0).checkState() == Qt.Checked

        item.setCheckState(Qt.Unchecked)
        assert settings.get_mod_art_included() == frozenset()
        assert not _draws_mod_green(window)
        assert "Mod art from 1_Green: off" in window.status_log.toPlainText()
    finally:
        for opened in (dialog, reopened):
            if opened is not None:
                opened.deleteLater()
        conftest.close_window(window)


def test_a_window_seeds_disabled_mod_opt_ins_before_any_panel(disabled_mod_art_profile) -> None:
    """GH #264, beside the opt-out seed test above: a persisted tick on a mod
    the game disables reaches the first render, sprite and textures; without
    the tick neither draws."""
    from PIL import Image

    from descape import mod_graphics

    textures = sprite_install_profile_textures(disabled_mod_art_profile)
    for name in set(asset_source._terrain_texture_map().values()):
        Image.new("RGB", (4, 4), (0, 250, 0)).save(textures / name, format="DDS")
    window = _mod_art_window("Stepped")
    try:
        assert mod_graphics.enabled() and not _draws_mod_green(window)
        for tid, name in asset_source._terrain_texture_map().items():
            assert asset_source.get_terrain_texture_path(tid) != textures / name, tid
    finally:
        conftest.close_window(window)

    settings.set_mod_art_included({disabled_mod_art_profile})
    window = _mod_art_window("Stepped")
    try:
        assert _draws_mod_green(window)
        for tid, name in asset_source._terrain_texture_map().items():
            assert asset_source.get_terrain_texture_path(tid) == textures / name, tid
    finally:
        conftest.close_window(window)


def _mods_dir():
    import os
    from pathlib import Path

    return Path(os.environ["AOE2DE_PROFILE_PATH"]) / "7656" / "mods"


def _set_in_game(**enabled: bool) -> None:
    """The game's mod-status.json listing these mods in this order, its mtime
    a whole second past the previous file's, so no timestamp granularity hides it."""
    import json
    import os

    status_file = _mods_dir() / "mod-status.json"
    before = status_file.stat().st_mtime_ns if status_file.exists() else None
    mods = [{"Enabled": on, "Path": f"subscribed//{name}", "Priority": n} for n, (name, on) in enumerate(enabled.items(), 1)]
    status_file.write_text(json.dumps({"Mods": mods}))
    if before is not None:
        ns = before + 1_000_000_000
        os.utime(status_file, ns=(ns, ns))


def _log(window) -> str:
    return window.status_log.toPlainText()


def _spy_refresh_map(window) -> list:
    calls = []
    real = window.refresh_map
    window.refresh_map = lambda: (calls.append(1), real())[1]
    return calls


PICKED_UP = "Picked up a change in the game's mod list"


def test_an_in_game_disable_waits_for_an_action_and_a_touch_is_nothing(mod_art_profile) -> None:
    """TASK-438: an in-game toggle changes nothing by itself (explicit-only,
    Decision 2); a pick-up redraws and says so, and the game's identical
    rewrite at launch neither logs nor re-renders."""
    window = _mod_art_window("Stepped")
    try:
        _set_in_game(**{"1_Green": False})
        assert _draws_mod_green(window), "no action, no change"
        assert window._pick_up_mod_changes()
        assert not _draws_mod_green(window)
        assert f"{PICKED_UP}. Mod art: none drawn" in _log(window)

        _set_in_game(**{"1_Green": True})
        assert window._pick_up_mod_changes()
        assert _draws_mod_green(window)

        logged = _log(window)
        renders = _spy_refresh_map(window)
        _set_in_game(**{"1_Green": True})  # the same list, rewritten
        assert not window._pick_up_mod_changes()
        assert renders == [] and _log(window) == logged
        _set_in_game(**{"1_Green": True})
        window.layer_actions[MOD_ART_LAYER].setChecked(False)
        assert renders == [1] and PICKED_UP not in _log(window)[len(logged):]
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("action", ["toggle", "open", "settings"])
def test_each_explicit_action_picks_up_an_in_game_disable(mod_art_profile, action, monkeypatch) -> None:
    from descape import mod_graphics
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = _mod_art_window("Stepped")
    try:
        assert _draws_mod_green(window)
        _set_in_game(**{"1_Green": False})
        logged = _log(window)
        if action == "toggle":
            window.layer_actions[MOD_ART_LAYER].setChecked(False)
            assert f"{PICKED_UP}. Mod Art: off" in _log(window)[len(logged):]
            window.layer_actions[MOD_ART_LAYER].setChecked(True)
            assert not _draws_mod_green(window)
        elif action == "open":
            # The same document again, unit included: the load's one render draws the pick-up.
            renders, prefetched = [], []
            real, real_prefetch = window._render_current, window._start_sprite_prefetch
            window._render_current = lambda **kw: (renders.append(1), real(**kw))[1]
            window._start_sprite_prefetch = lambda: (prefetched.append(mod_graphics.generation()), real_prefetch())[1]
            window.load_scenario(BLANK_TEMPLATE_PATH, loaded=window.scenario)
            assert PICKED_UP in _log(window)[len(logged):]
            assert renders == [1] and not _draws_mod_green(window)
            assert prefetched == [mod_graphics.generation()], "a bump after the prefetch discards its work"
        else:
            monkeypatch.setattr(SettingsDialog, "exec_", lambda self: 0)
            window._show_settings()
            assert PICKED_UP in _log(window)[len(logged):]
            assert not _draws_mod_green(window)
        assert mod_graphics.resolve_sprite(FILE_NAME) is None
    finally:
        conftest.close_window(window)


def test_with_mod_art_off_a_pick_up_redraws_nothing_and_settings_lists_the_new_state(mod_art_profile) -> None:
    from PyQt5.QtCore import Qt

    window = _mod_art_window("Stepped")
    dialog = None
    try:
        action = window.layer_actions[MOD_ART_LAYER]
        action.setChecked(False)
        _set_in_game(**{"1_Green": False})
        renders = _spy_refresh_map(window)
        assert window._pick_up_mod_changes()
        assert renders == [] and _log(window).rstrip().endswith(PICKED_UP)
        dialog = SettingsDialog(window)
        item = dialog.mod_art_list.item(0)
        assert item.text() == "1_Green (disabled in game)" and item.checkState() == Qt.Unchecked
        assert action.isEnabled(), "a disabled mod's art still counts for the row"
    finally:
        if dialog is not None:
            dialog.deleteLater()
        conftest.close_window(window)


def test_an_opt_out_with_settings_open_relists_a_mod_disabled_meanwhile(mod_art_profile) -> None:
    """Settings is modal, so a mod toggled in the game while it is open shows
    on the next tick there, in a deferred refresh; the path field is left alone."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    graphics = _mods_dir() / "subscribed" / "2_Blue" / "resources" / "_common" / "drs" / "graphics"
    graphics.mkdir(parents=True)
    (graphics / "other_x1.smx").write_bytes(b"x")
    _set_in_game(**{"1_Green": True, "2_Blue": True})
    window = _mod_art_window("Stepped")
    dialog = SettingsDialog(window)
    try:
        dialog.profile_path_edit.setText("half typed")
        rows = {dialog.mod_art_list.item(i).data(Qt.UserRole): i for i in range(dialog.mod_art_list.count())}
        _set_in_game(**{"1_Green": False, "2_Blue": True})
        dialog.mod_art_list.item(rows["subscribed//2_Blue"]).setCheckState(Qt.Unchecked)
        assert settings.get_mod_art_excluded() == {"subscribed//2_Blue"}
        QApplication.processEvents()
        texts = {dialog.mod_art_list.item(i).data(Qt.UserRole): dialog.mod_art_list.item(i) for i in range(2)}
        green = texts[mod_art_profile]
        assert green.text() == "1_Green (disabled in game)" and green.checkState() == Qt.Unchecked
        assert texts["subscribed//2_Blue"].checkState() == Qt.Unchecked
        assert dialog.profile_path_edit.text() == "half typed"
        assert not _draws_mod_green(window)
    finally:
        dialog.deleteLater()
        conftest.close_window(window)


def test_a_busy_window_picks_up_nothing(mod_art_profile) -> None:
    from descape import mod_graphics

    window = _mod_art_window("Stepped")
    try:
        _set_in_game(**{"1_Green": False})
        before = mod_graphics.generation()
        window._busy = True
        try:
            assert not window._pick_up_mod_changes()
        finally:
            window._busy = False
        assert mod_graphics.generation() == before
    finally:
        conftest.close_window(window)


def test_rescan_mods_redraws_a_mod_the_game_updated_in_place(mod_art_profile) -> None:
    """The signature cannot see a mod's own folder change (the game updated
    the mod), so only Rescan mods re-reads it: the update ships the stem as
    an .sld too, which beats the green .smx inside the mod and draws like
    the install's (playercolour-tinted, never green)."""
    window = _mod_art_window("Stepped")
    dialog = SettingsDialog(window)
    try:
        assert _draws_mod_green(window)
        graphics = _mods_dir() / "subscribed" / "1_Green" / "resources" / "_common" / "drs" / "graphics"
        (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4))
        assert not window._pick_up_mod_changes()
        assert _draws_mod_green(window)
        dialog.mod_rescan_button.click()
        assert not _draws_mod_green(window)
        assert "Rescanned mods" in _log(window)
    finally:
        dialog.deleteLater()
        conftest.close_window(window)


def test_an_ai_mod_enabled_in_the_game_joins_personality_on_a_pick_up(mod_art_profile) -> None:
    ai = _mods_dir() / "subscribed" / "3_Ai" / "resources" / "_common" / "ai"
    ai.mkdir(parents=True)
    (ai / "ModAi.ai").write_bytes(b"")
    (ai / "ModAi.per").write_bytes(b"(defrule)")
    _set_in_game(**{"1_Green": True, "3_Ai": False})
    window = _mod_art_window("Stepped")
    try:
        assert "ModAi.ai" not in {c.stored_name for c in window._ai_scan()}
        _set_in_game(**{"1_Green": True, "3_Ai": True})
        renders = _spy_refresh_map(window)
        assert window._pick_up_mod_changes()
        assert "ModAi.ai" in {c.stored_name for c in window._ai_scan()}
        assert renders == [] and _draws_mod_green(window), "a mod without art redraws nothing"
    finally:
        conftest.close_window(window)


def test_a_profile_change_rescans_the_mod_folders(mod_art_profile, tmp_path, monkeypatch) -> None:
    window = _mod_art_window("Stepped")
    dialog = SettingsDialog(window)
    try:
        assert _draws_mod_green(window)
        empty = tmp_path / "empty_profile"
        (empty / "7656").mkdir(parents=True)
        monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(empty))
        dialog.use_auto_profile_path()
        assert not _draws_mod_green(window)
        assert dialog.mod_art_list.item(0).data(0x0100) is None  # Qt.UserRole: the placeholder row
        assert dialog.mod_art_list.item(0).text() == "No mod in the profile folder above ships art."
    finally:
        dialog.deleteLater()
        conftest.close_window(window)
