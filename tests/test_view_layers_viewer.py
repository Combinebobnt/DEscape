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

from descape import asset_source, unit_sprites, view_layers, viewer

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
                    sprites_enabled=window._sprites_enabled, has_map=True,
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
    Flat renders through IsoChunkCache and the farm layer stays live, so the
    greying half of this test would never fire."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Stepped")
        window.layer_actions[FARM_LAYER].setChecked(False)
        assert window._layers.farm_overlay is False

        window.terrain_style_combo.setCurrentText("Flat")
        window.iso_action.setChecked(False)
        assert window._render_style == "flat", "fixture no longer reaches real Flat"
        assert not window.layer_actions[FARM_LAYER].isEnabled()
        assert window.layer_actions[FARM_LAYER].isChecked() is False
        assert window._layers.farm_overlay is False

        window.terrain_style_combo.setCurrentText("Stepped")
        assert window.layer_actions[FARM_LAYER].isEnabled()
        assert window.layer_actions[FARM_LAYER].isChecked() is False
        assert window._cache.layers.farm_overlay is False
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
        monkeypatch.setattr(
            viewer, "unit_sprite_draws_at",
            lambda *args, **kwargs: (seen.append(kwargs["tree_scale"]), [])[1],
        )
        monkeypatch.setattr(type(window), "_ghost_rotation_override", lambda self, entry: None)
        ghost = SimpleNamespace(x=5.5, y=5.5, unit_const=CONST, rotation=0.0, reference_id=1)
        entry = SimpleNamespace(player_id=1, unit=ghost, order=0)

        window.layer_actions[SMALL_TREES_LAYER].setChecked(True)
        assert window._ghost_sprite_draws(entry, ghost) == []
        assert seen == [view_layers.SMALL_TREE_SCALE]

        window.layer_actions[SMALL_TREES_LAYER].setChecked(False)
        window._ghost_sprite_draws(entry, ghost)
        assert seen[-1] == 1.0
    finally:
        conftest.close_window(window)
