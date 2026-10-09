"""Settings > Appearance's per-element tool overlay color grid --
descape.viewer.SettingsDialog's swatch rows and MapView.apply_overlay_colors()'s
live-push path. Same offscreen-ViewerWindow technique tests/test_fill_tool.py
documents: QT_QPA_PLATFORM=offscreen, one shared QApplication via
conftest.ensure_qapp(), and every window edit_history.mark_saved()'d before
close() so closeEvent's discard prompt can't block forever offscreen.

SettingsDialog is constructed directly rather than through
ViewerWindow._show_settings() (that method calls exec_(), which is modal and
would hang an offscreen run) -- same precedent as tests/test_keybinds.py and
tests/test_elev_step_slider.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_ruler_viewer import _drag, _pending, _ruler_window

from descape import settings
from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


_dialog_and_window = conftest.dialog_and_window


def test_every_overlay_color_prefix_has_a_section_title() -> None:
    """Reflective, mirroring test_settings.py's memoized-globals check: an
    unlisted prefix still renders (falls back to section.title()), so
    nothing would fail loudly if this decayed on its own -- the Appearance
    tab would just grow an ungrouped, oddly-cased header."""
    from descape.viewer import SettingsDialog

    prefixes = {color_id.split("_", 1)[0] for color_id, _label, _default in settings.OVERLAY_COLORS}
    assert prefixes <= set(SettingsDialog._OVERLAY_SECTION_TITLES)


def test_every_overlay_color_has_a_matching_swatch_row() -> None:
    """Both directions, mirroring test_keybinds.py's test_every_rebindable_
    action_has_a_matching_keybind_action: a missing row leaves that color
    permanently stuck at whatever it happens to be, and a stray row would be
    dead UI for an id nothing else recognizes."""
    dialog, window = _dialog_and_window()
    try:
        declared_ids = {color_id for color_id, _label, _default in settings.OVERLAY_COLORS}
        assert declared_ids <= set(dialog._overlay_swatches)
        assert set(dialog._overlay_swatches) <= declared_ids
    finally:
        dialog.close()
        window.edit_history.mark_saved()
        window.close()


def test_picking_a_ruler_color_paints_the_live_measurement(tmp_path, monkeypatch) -> None:
    """The real proof, not just that settings.py round-trips: pin ruler_line
    to a distinctive magenta before the window opens, draw a measurement on
    uniform terrain, and check the actual rendered pixels -- both that the
    configured color shows up AND that the old default doesn't linger
    anywhere (e.g. a class-constant draw site that got missed)."""
    from PyQt5.QtGui import QImage

    from testkit.qt_capture import qimage_rgb888_to_array

    (tmp_path / "config.yaml").write_text("overlay_colors:\n  ruler_line: '#ff00ff'\n")
    monkeypatch.setattr(settings, "_overlay_colors", None)

    window = _ruler_window()
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (30, 24))
        pixmap = map_view.viewport().grab()
        array = qimage_rgb888_to_array(pixmap.toImage().convertToFormat(QImage.Format_RGB888))

        magenta = (255, 0, 255)
        default_orange = tuple(
            int(settings.get_default_overlay_color("ruler_line")[i : i + 2], 16) for i in (1, 3, 5)
        )
        hit_magenta = (np.abs(array.astype(np.int16) - np.array(magenta, dtype=np.int16)) <= 8).all(axis=2)
        hit_default = (np.abs(array.astype(np.int16) - np.array(default_orange, dtype=np.int16)) <= 8).all(
            axis=2
        )
        assert hit_magenta.any(), "configured ruler color never appeared in the capture"
        assert not hit_default.any(), "default ruler color still showing despite the override"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_live_apply_repaints_without_replacing_the_item(tmp_path, monkeypatch) -> None:
    """apply_overlay_colors() must update the existing scene item's pen/brush
    in place, not remove-and-recreate it -- a recreate would drop whatever
    else (z-order, event filters) the original item carried."""
    from PyQt5.QtGui import QColor

    window = _ruler_window()
    try:
        map_view = window.map_view
        _pending(map_view, (10, 10), (30, 24))
        item_id_before = id(map_view._ruler_line_item)

        settings.set_overlay_color("ruler_line", "#ff00ff")
        map_view.apply_overlay_colors()

        assert id(map_view._ruler_line_item) == item_id_before
        assert map_view._ruler_line_item.pen().color() == QColor("#ff00ff")
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- overlay opacity (GH #129) -----------------------------------------------


def _opacities(items) -> list[float]:
    return [round(item.opacity(), 3) for item in items]


def _selection_items(map_view) -> list:
    return [item for group in map_view._unit_select_groups.values() for item in group if item is not None]


def _region_items(map_view) -> list:
    return [map_view._region_fill_item, map_view._region_outline_item, map_view._region_ants_item]


def _units_window(by_owner: bool = True):
    """test_unit_selection_viewer's GAIA tree, P1 and P2 buildings, in Units mode."""
    from test_unit_selection_viewer import _owner_window

    settings.set_selection_by_owner(by_owner)
    return _owner_window()


def _select_all_units(window) -> None:
    window.map_view.set_unit_selection(list(window.map_view._unit_index.entries))


def _start_marquee(map_view) -> None:
    from PyQt5.QtCore import QPoint

    map_view._update_marquee(QPoint(5, 5), QPoint(60, 40))


def test_every_overlay_opacity_group_has_exactly_one_slider_row() -> None:
    """Each group sits under a colour section that exists, and its slider is
    not a swatch: test_every_overlay_color_has_a_matching_swatch_row pins those."""
    from PyQt5.QtWidgets import QSlider

    from descape.viewer import SettingsDialog

    group_ids = [gid for gid, _label, _default in settings.OVERLAY_OPACITIES]
    prefixes = {color_id.split("_", 1)[0] for color_id, _label, _default in settings.OVERLAY_COLORS}
    assert sorted(SettingsDialog._OVERLAY_OPACITY_SECTIONS.values()) == sorted(group_ids)
    assert set(SettingsDialog._OVERLAY_OPACITY_SECTIONS) <= prefixes
    dialog, window = _dialog_and_window()
    try:
        assert sorted(dialog._overlay_opacity_sliders) == sorted(group_ids)
        named = [s for s in dialog.findChildren(QSlider) if s.objectName().startswith("overlay_opacity_")]
        assert sorted(s.objectName() for s in named) == sorted(f"overlay_opacity_{gid}" for gid in group_ids)
        for gid in group_ids:
            slider = dialog._overlay_opacity_sliders[gid]
            assert (slider.minimum(), slider.maximum()) == (settings.OVERLAY_OPACITY_MIN, settings.OVERLAY_OPACITY_MAX)
            assert slider.value() == settings.get_overlay_opacity(gid)
        assert set(dialog._overlay_swatches) == {cid for cid, _label, _default in settings.OVERLAY_COLORS}
    finally:
        dialog.close()
        conftest.close_window(window)


def test_apply_overlay_opacity_fades_every_selection_and_region_item_in_place() -> None:
    window = _units_window()
    try:
        mv = window.map_view
        _select_all_units(window)
        _start_marquee(mv)
        mv.set_region((2, 2, 6, 6))
        items = [*_selection_items(mv), mv._marquee_item, *_region_items(mv)]
        assert len(_selection_items(mv)) == 9, "three owner groups, each fill, under-stroke and outline"
        assert set(_opacities(items)) == {1.0}
        ids_before = [id(item) for item in items]

        settings.set_overlay_opacity("unit_select", 40)
        settings.set_overlay_opacity("region", 30)
        mv.apply_overlay_opacity()

        assert [id(item) for item in [*_selection_items(mv), mv._marquee_item, *_region_items(mv)]] == ids_before
        assert set(_opacities([*_selection_items(mv), mv._marquee_item])) == {0.4}
        assert set(_opacities(_region_items(mv))) == {0.3}
        # A multiplier on the item, never the colour.
        fill, _under, _outline = mv._unit_select_groups[None]
        assert fill.brush().color().alpha() == mv.UNIT_SELECT_FILL_ALPHA
        assert mv._region_fill_item.brush().color().alpha() == mv.REGION_SELECT_FILL_ALPHA
    finally:
        conftest.close_window(window)


def test_selections_created_after_the_change_start_faded() -> None:
    settings.set_overlay_opacity("unit_select", 40)
    settings.set_overlay_opacity("region", 30)
    window = _units_window()
    try:
        mv = window.map_view
        _select_all_units(window)
        _start_marquee(mv)
        mv.set_region((2, 2, 6, 6))
        assert set(_opacities([*_selection_items(mv), mv._marquee_item])) == {0.4}
        assert set(_opacities(_region_items(mv))) == {0.3}
    finally:
        conftest.close_window(window)


def test_turning_colour_by_owner_on_fades_the_new_under_stroke() -> None:
    """GAIA keeps the configured group key either way, so the toggle adds an
    under-stroke to a live group rather than building a new one."""
    window = _units_window(by_owner=False)
    try:
        mv = window.map_view
        gaia = next(e for e in mv._unit_index.entries if e.player_id == 0)
        mv.set_unit_selection([gaia])
        assert mv._unit_select_groups[None][1] is None
        settings.set_overlay_opacity("unit_select", 40)
        mv.apply_overlay_opacity()
        mv.set_selection_by_owner(True)
        under = mv._unit_select_groups[None][1]
        assert under is not None
        assert round(under.opacity(), 3) == 0.4
    finally:
        conftest.close_window(window)


def test_a_live_override_reaches_a_group_created_mid_drag_and_clears_on_persist() -> None:
    window = _units_window()
    try:
        mv = window.map_view
        mv.apply_overlay_opacity(overrides={"unit_select": 25})
        assert settings.get_overlay_opacity("unit_select") == 100, "an override is never persisted"
        _select_all_units(window)
        assert set(_opacities(_selection_items(mv))) == {0.25}
        mv.apply_overlay_opacity()
        assert set(_opacities(_selection_items(mv))) == {1.0}
    finally:
        conftest.close_window(window)


def _trigger_window():
    from test_trigger_overlay_viewer import PATROL, _window

    window = _window()
    window.trigger_panel.select_entry(*PATROL)
    return window


def _trigger_roles(map_view) -> dict:
    return {item.data(0): item for item in map_view.trigger_overlay_items()}


def test_trigger_area_opacity_fades_areas_only_and_keeps_the_items() -> None:
    window = _trigger_window()
    try:
        mv = window.map_view
        roles = _trigger_roles(mv)
        assert {"fill_strong", "outline_strong", "fill_dim", "outline_dim", "label"} <= set(roles)
        ids_before = sorted(id(item) for item in mv.trigger_overlay_items())

        settings.set_overlay_opacity("trigger_area", 40)
        mv.apply_overlay_opacity()

        assert sorted(id(item) for item in mv.trigger_overlay_items()) == ids_before
        for role, item in _trigger_roles(mv).items():
            faded = role.startswith(("fill_", "outline_"))
            assert round(item.opacity(), 3) == (0.4 if faded else 1.0), role
        assert roles["fill_strong"].brush().color().alpha() == mv.TRIGGER_FILL_ALPHA
        assert roles["fill_dim"].brush().color().alpha() == mv.TRIGGER_DIM_FILL_ALPHA
    finally:
        conftest.close_window(window)


def test_a_trigger_colour_change_keeps_the_area_opacity() -> None:
    """apply_overlay_colors rebuilds the trigger items, so the creation site has to carry it."""
    window = _trigger_window()
    try:
        mv = window.map_view
        settings.set_overlay_opacity("trigger_area", 40)
        mv.apply_overlay_opacity()
        settings.set_overlay_color("trigger_area_outline", "#ff00ff")
        mv.apply_overlay_colors()
        roles = _trigger_roles(mv)
        assert roles["outline_strong"].pen().color().name() == "#ff00ff"
        for role in ("fill_strong", "outline_strong", "fill_dim", "outline_dim"):
            assert round(roles[role].opacity(), 3) == 0.4, role
        assert round(roles["label"].opacity(), 3) == 1.0
    finally:
        conftest.close_window(window)


def test_apply_overlay_opacity_with_no_map_and_after_close_is_a_no_op() -> None:
    window = conftest.blank_window(load=False)
    try:
        window.map_view.apply_overlay_opacity()
        window.map_view.apply_overlay_opacity(overrides={"region": 20})
    finally:
        conftest.close_window(window)
    window = _units_window()
    try:
        _select_all_units(window)
        window.map_view.set_region((2, 2, 6, 6))
        window.close_scenario()
        window.map_view.apply_overlay_opacity()
        window.map_view.apply_overlay_opacity(overrides={"unit_select": 20, "region": 20, "trigger_area": 20})
    finally:
        conftest.close_window(window)


def test_the_slider_fades_live_and_persists_once_on_close() -> None:
    from descape.viewer import SettingsDialog

    window = conftest.blank_window()
    dialog = SettingsDialog(window)
    try:
        mv = window.map_view
        mv.set_region((2, 2, 6, 6))
        dialog._overlay_opacity_sliders["region"].setValue(40)
        assert set(_opacities(_region_items(mv))) == {0.4}
        assert dialog._overlay_opacity_labels["region"].text() == "40%"
        # Debounced: every setter is a full YAML round trip.
        assert settings.get_overlay_opacity("region") == 100
        dialog.done(0)
        assert settings.get_overlay_opacity("region") == 40
        assert "Overlay opacity: Select tool -> 40%" in window.status_log.toPlainText()
    finally:
        dialog.close()
        conftest.close_window(window)


def test_the_opacity_default_button_restores_and_persists_at_once() -> None:
    from descape.viewer import SettingsDialog

    settings.set_overlay_opacity("trigger_area", 30)
    window = conftest.blank_window()
    dialog = SettingsDialog(window)
    try:
        assert dialog._overlay_opacity_sliders["trigger_area"].value() == 30
        dialog._overlay_opacity_defaults["trigger_area"].click()
        assert dialog._overlay_opacity_sliders["trigger_area"].value() == 100
        assert settings.get_overlay_opacity("trigger_area") == 100
    finally:
        dialog.close()
        conftest.close_window(window)


def test_apply_overlay_colors_clears_the_stale_edit_highlight() -> None:
    """_clear_highlight() runs as part of apply_overlay_colors() (the edit
    highlight has no in-place update path) -- confirm the highlight actually
    goes away rather than keep showing the old color until the next hover."""
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    try:
        map_view = window.map_view
        window._on_tool_selected("draw")
        map_view._update_highlight(5, 5)
        assert map_view._highlight_outline_item is not None

        settings.set_overlay_color("highlight_outline", "#123456")
        map_view.apply_overlay_colors()

        assert map_view._highlight_outline_item is None
        assert map_view._highlight_key is None
    finally:
        window.edit_history.mark_saved()
        window.close()
