"""The visual terrain picker's Qt wiring: swatches, grouping and filtering in
TerrainPickerView, and the Terrain-mode sidebar page (TerrainPanel, GH #56)
that is now the only place a terrain is chosen.

Same offscreen technique tests/test_fill_tool.py documents. conftest.py's
autouse _isolated_settings hides the configured AoE2DE install, so unless a
test asks for one these run down the no-install path -- which is the path
worth pinning anyway: every swatch must still be a real flat-colour pixmap,
never a blank.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

from descape import asset_source, terrain_browser, terrain_catalog
from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_GRASS_1 = 0
_SNOW_SOFT = next(
    e.id for e in terrain_catalog.terrains() if e.name == "SNOW_SOFT"
)
_BLACK = next(e.id for e in terrain_catalog.terrains() if e.name == "BLACK")
_WATER_DEEP = 22


def _view(current_id: int | None = _GRASS_1):
    conftest.ensure_qapp()
    return terrain_browser.TerrainPickerView(current_id)


def _rows(tree):
    for i in range(tree.topLevelItemCount()):
        group = tree.topLevelItem(i)
        for c in range(group.childCount()):
            yield group, group.child(c)


def _row_for(tree, value):
    return next(row for _group, row in _rows(tree) if row.data(0, _user_role()) == value)


def _terrain_window():
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("draw")
    # Trees default on and would pop an unpatched large-edit QMessageBox below, hanging offscreen.
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    QApplication.processEvents()
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _user_pick(window, value) -> None:
    """What a click on a row does: moves the tree's current item, nothing else."""
    tree = window.terrain_panel.view.tree
    tree.setCurrentItem(_row_for(tree, value))


# -- the picker view ---------------------------------------------------------


def test_every_terrain_appears_under_exactly_one_group() -> None:
    view = _view()
    try:
        values = [row.data(0, _user_role()) for _group, row in _rows(view.tree)]
        assert sorted(values) == sorted(e.id for e in terrain_catalog.terrains())
    finally:
        view.close()


def _user_role():
    from PyQt5.QtCore import Qt

    return Qt.UserRole


def test_the_junk_terrains_are_hidden_until_opted_into() -> None:
    view = _view()
    try:
        hidden_ids = {e.id for e in terrain_catalog.terrains() if e.hidden}
        shown = {
            row.data(0, _user_role())
            for _group, row in _rows(view.tree)
            if not row.isHidden()
        }
        assert not (shown & hidden_ids)
        view.show_hidden_checkbox.setChecked(True)
        shown = {
            row.data(0, _user_role())
            for _group, row in _rows(view.tree)
            if not row.isHidden()
        }
        assert hidden_ids <= shown
    finally:
        view.close()


def test_it_opens_on_the_current_terrains_category_only() -> None:
    """Load-bearing for open cost, not just for convenience: with no
    default_group every group expands, and an all-expanded tree decodes all
    85 distinct textures at once (measured at ~2.8s cold)."""
    view = _view(_SNOW_SOFT)
    try:
        expanded = [
            view.tree.topLevelItem(i).text(0)
            for i in range(view.tree.topLevelItemCount())
            if view.tree.topLevelItem(i).isExpanded()
        ]
        assert len(expanded) == 1
        assert expanded[0].startswith("Snow & Ice")
        assert view.tree.currentItem().data(0, _user_role()) == _SNOW_SOFT
    finally:
        view.close()


def test_expanded_rows_carry_a_swatch_icon_with_no_install() -> None:
    """The no-install path must still show colour, not blanks -- that is the
    degradation contract color_for_terrain_id exists for."""
    view = _view(_GRASS_1)
    try:
        assert not asset_source.is_available(), "conftest should hide the install here"
        expanded_rows = [
            row for group, row in _rows(view.tree) if group.isExpanded()
        ]
        assert expanded_rows
        for row in expanded_rows:
            assert not row.icon(0).isNull(), row.text(0)
    finally:
        view.close()


def test_collapsed_rows_stay_undecoded_until_expanded() -> None:
    view = _view(_GRASS_1)
    try:
        collapsed = [
            (group, row) for group, row in _rows(view.tree) if not group.isExpanded()
        ]
        assert collapsed
        group, row = collapsed[0]
        assert row.icon(0).isNull()
        group.setExpanded(True)
        assert not row.icon(0).isNull()
    finally:
        view.close()


def test_filtering_reveals_rows_with_their_icons_populated() -> None:
    """itemExpanded fires on _apply_filter's own programmatic setExpanded,
    which is what covers a filter-revealed row -- verified here rather than
    assumed, since a silent failure would mean iconless rows exactly when
    the user is searching."""
    view = _view(_GRASS_1)
    try:
        view.filter_edit.setText("snow")
        revealed = [row for _group, row in _rows(view.tree) if not row.isHidden()]
        assert revealed
        assert all("snow" in row.text(0).lower() for row in revealed)
        assert all(not row.icon(0).isNull() for row in revealed)
    finally:
        view.close()


def test_the_raw_enum_name_is_searchable_too() -> None:
    """The display label is title-cased with spaces, so filtering on the
    underscored enum name would otherwise miss."""
    view = _view(_GRASS_1)
    try:
        view.filter_edit.setText("non_navigable")
        revealed = [row for _group, row in _rows(view.tree) if not row.isHidden()]
        assert revealed
    finally:
        view.close()


def test_each_swatch_shows_its_own_terrains_colour() -> None:
    """Measure, don't eyeball: a browser of 131 identical squares would sail
    through a "did it render" check. Asserted as "the swatch matches this
    terrain's own palette colour" rather than "every category differs" --
    the offscreen platform rounds a filled pixmap's channels by a few
    counts, which collapses two near-neighbour categories into one value
    and says nothing about the swatch being right."""
    from descape import terrain_palette

    view = _view(_GRASS_1)
    try:
        by_category = {}
        for entry in terrain_catalog.terrains():
            by_category.setdefault(entry.category, entry.id)
        seen = set()
        for terrain_id in by_category.values():
            image = terrain_browser.swatch_pixmap(terrain_id, 8).toImage()
            pixel = image.pixel(4, 4)
            seen.add(pixel)
            shown = ((pixel >> 16) & 0xFF, (pixel >> 8) & 0xFF, pixel & 0xFF)
            expected = terrain_palette.color_for_terrain_id(terrain_id)
            assert all(abs(a - b) <= 8 for a, b in zip(shown, expected, strict=True)), (
                terrain_id,
                shown,
                expected,
            )
        # And they are not one repeated square.
        assert len(seen) >= len(by_category) - 2
    finally:
        view.close()


def test_refresh_swatches_replaces_existing_icons(monkeypatch) -> None:
    """The view is long-lived now, and _populate_icons skips any row that
    already has an icon, so without the clear an install change would leave
    the old swatches up forever."""
    from PyQt5.QtGui import QColor, QPixmap

    view = _view(_GRASS_1)
    try:
        row = next(row for group, row in _rows(view.tree) if group.isExpanded())
        before = row.icon(0).pixmap(8, 8).toImage().pixel(4, 4)

        def _magenta(_terrain_id, px=terrain_browser.SWATCH_PX):
            pixmap = QPixmap(px, px)
            pixmap.fill(QColor(255, 0, 255))
            return pixmap

        monkeypatch.setattr(terrain_browser, "swatch_pixmap", _magenta)
        view.refresh_swatches()
        after = row.icon(0).pixmap(8, 8).toImage().pixel(4, 4)
        assert after != before
        assert (after & 0xFFFFFF) == 0xFF00FF
    finally:
        view.close()


# -- the Terrain-mode sidebar page -------------------------------------------


def test_terrain_mode_shows_the_picker_page_and_view_does_not() -> None:
    from descape.viewer import _LEFT_PAGE_INFO, _LEFT_PAGE_TERRAIN

    window = _terrain_window()
    try:
        assert window.left_stack.currentIndex() == _LEFT_PAGE_TERRAIN == 7
        assert window.left_stack.currentWidget() is window.terrain_panel
        window.mode_combo.setCurrentText("View")
        assert window.left_stack.currentIndex() == _LEFT_PAGE_INFO == 0
    finally:
        _close(window)


def test_entering_terrain_widens_a_too_narrow_pane() -> None:
    from descape.terrain_panel import TerrainPanel

    window = _terrain_window()
    try:
        window.mode_combo.setCurrentText("View")
        window.content_splitter.setSizes([200, 1000])
        window.mode_combo.setCurrentText("Terrain")
        assert window.content_splitter.sizes()[0] >= TerrainPanel.MIN_USEFUL_WIDTH
    finally:
        _close(window)


def test_the_toolbar_terrain_combo_and_browse_button_are_gone() -> None:
    from PyQt5.QtWidgets import QLabel, QPushButton, QToolBar

    window = _terrain_window()
    try:
        assert not hasattr(window, "terrain_combo")
        assert not hasattr(window, "terrain_browse_button")
        toolbars = window.findChildren(QToolBar)
        assert toolbars
        for toolbar in toolbars:
            assert not any("Terrain type" in label.text() for label in toolbar.findChildren(QLabel))
            assert not any(button.text() == "…" for button in toolbar.findChildren(QPushButton))
    finally:
        _close(window)


def test_the_default_terrain_is_the_retired_combos_first_entry() -> None:
    """The combo listed TerrainId sorted by enum name and started on index 0;
    moving the picker must not change what an untouched Draw stroke paints."""
    from AoE2ScenarioParser.datasets.terrains import TerrainId

    window = _terrain_window()
    try:
        assert window.terrain_panel.terrain_id() == sorted(TerrainId, key=lambda t: t.name)[0].value
    finally:
        _close(window)


def test_a_picked_row_is_what_draw_paint_can_and_shapes_paint() -> None:
    window = _terrain_window()
    try:
        mm = window.scenario.map_manager
        _user_pick(window, _SNOW_SOFT)
        assert window.terrain_panel.terrain_id() == _SNOW_SOFT

        window._on_tool_selected("draw")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(3, 3, 0)
        window.on_edit_stroke_end()
        assert mm.get_tile(3, 3).terrain_id == _SNOW_SOFT

        _user_pick(window, _WATER_DEEP)
        window._on_tool_selected("draw_rect")
        window.on_shape_commit([(8, 8), (9, 8)])
        assert mm.get_tile(8, 8).terrain_id == _WATER_DEEP
        assert mm.get_tile(9, 8).terrain_id == _WATER_DEEP

        _user_pick(window, _GRASS_1)
        window._on_tool_selected("fill")
        window.on_fill(3, 3, 0)
        assert mm.get_tile(3, 3).terrain_id == _GRASS_1
    finally:
        _close(window)


def test_a_group_heading_click_keeps_the_previous_terrain() -> None:
    window = _terrain_window()
    try:
        _user_pick(window, _SNOW_SOFT)
        tree = window.terrain_panel.view.tree
        tree.setCurrentItem(tree.topLevelItem(0))
        assert window.terrain_panel.terrain_id() == _SNOW_SOFT
        # A filter that hides the current row keeps it too.
        window.terrain_panel.view.filter_edit.setText("desert")
        assert window.terrain_panel.terrain_id() == _SNOW_SOFT
    finally:
        _close(window)


def test_a_user_pick_regates_auto_beach() -> None:
    window = _terrain_window()
    try:
        _user_pick(window, _GRASS_1)
        assert not window.auto_beach_check.isEnabled()
        _user_pick(window, _WATER_DEEP)
        assert window.auto_beach_check.isEnabled()
    finally:
        _close(window)


def test_the_eyedropper_selects_the_row_in_the_panel() -> None:
    window = _terrain_window()
    try:
        window.scenario.map_manager.terrain[0].terrain_id = _SNOW_SOFT
        window.pick_tile_value(0, 0, 0)
        assert window.terrain_panel.terrain_id() == _SNOW_SOFT
        assert window.terrain_panel.view.current_value() == _SNOW_SOFT
    finally:
        _close(window)


def test_the_eyedropper_reveals_a_hidden_terrain() -> None:
    window = _terrain_window()
    try:
        view = window.terrain_panel.view
        assert not view.show_hidden_checkbox.isChecked()
        window.scenario.map_manager.terrain[0].terrain_id = _BLACK
        window.pick_tile_value(0, 0, 0)
        assert view.show_hidden_checkbox.isChecked()
        assert window.terrain_panel.terrain_id() == _BLACK
        assert view.current_value() == _BLACK
        assert not view.tree.currentItem().isHidden()
    finally:
        _close(window)


def test_the_eyedropper_clears_only_a_filter_that_hides_the_pick() -> None:
    window = _terrain_window()
    try:
        view = window.terrain_panel.view
        mm = window.scenario.map_manager

        view.filter_edit.setText("snow")
        mm.terrain[0].terrain_id = _SNOW_SOFT
        window.pick_tile_value(0, 0, 0)
        assert view.filter_edit.text() == "snow", "a filter that shows the row is the user's; keep it"
        assert view.current_value() == _SNOW_SOFT

        mm.terrain[0].terrain_id = _WATER_DEEP
        window.pick_tile_value(0, 0, 0)
        assert view.filter_edit.text() == ""
        assert window.terrain_panel.terrain_id() == _WATER_DEEP
        assert view.current_value() == _WATER_DEEP
    finally:
        _close(window)


def test_a_non_catalog_id_leaves_the_panel_unchanged() -> None:
    window = _terrain_window()
    try:
        _user_pick(window, _SNOW_SOFT)
        assert window.terrain_panel.set_terrain(9999) is False
        assert window.terrain_panel.terrain_id() == _SNOW_SOFT
        assert window.terrain_panel.view.current_value() == _SNOW_SOFT
    finally:
        _close(window)


def test_set_terrain_emits_once_and_never_for_a_no_change() -> None:
    window = _terrain_window()
    try:
        panel = window.terrain_panel
        _user_pick(window, _GRASS_1)
        emitted: list[int] = []
        panel.terrain_changed.connect(emitted.append)
        assert panel.set_terrain(_BLACK)  # hidden: ticks show-hidden and expands a group on the way
        assert emitted == [_BLACK]
        assert panel.set_terrain(_BLACK)
        assert emitted == [_BLACK]
    finally:
        _close(window)


def test_the_hover_readout_reaches_the_panel_line() -> None:
    from descape.viewer import HOVER_IDLE_TEXT

    window = _terrain_window()
    try:
        panel = window.terrain_panel
        assert panel.hover_text() == HOVER_IDLE_TEXT
        window.on_hover((2, 3))
        assert panel.hover_text().startswith("(2, 3)")
        assert panel.hover_text() == window.hover_label.text()
        assert not panel.hover_label.wordWrap()
        window.on_hover(None)
        assert panel.hover_text() == HOVER_IDLE_TEXT
    finally:
        _close(window)


def test_the_tree_takes_focus_only_on_click() -> None:
    """An always-visible tree with the default focus policy would eat
    MapView's arrow keys the moment it gained focus."""
    from PyQt5.QtCore import Qt

    window = _terrain_window()
    try:
        assert window.terrain_panel.view.tree.focusPolicy() == Qt.ClickFocus
    finally:
        _close(window)


def test_loading_an_install_path_refreshes_the_swatches(monkeypatch) -> None:
    dialog, window = conftest.dialog_and_window()
    try:
        calls: list[bool] = []
        monkeypatch.setattr(asset_source, "validate_install_path", lambda path: (True, "ok"))
        monkeypatch.setattr(asset_source, "set_install_path_override", lambda path: None)
        monkeypatch.setattr(asset_source, "save_install_path_to_config", lambda path: None)
        monkeypatch.setattr(window.terrain_panel.view, "refresh_swatches", lambda: calls.append(True))
        dialog.install_path_edit.setText("/nonexistent/aoe2de")
        dialog.load_install_path()
        assert calls == [True]
    finally:
        dialog.close()
        _close(window)


# -- cache registration ------------------------------------------------------


def test_changing_the_install_clears_the_thumbnail_cache() -> None:
    """The plan's own "this will bite" item: miss the clear list and the
    browser shows stale swatches after a GUI install change."""
    asset_source.get_terrain_thumbnail(_GRASS_1, 8)
    asset_source.set_install_path_override(None)
    assert asset_source._terrain_thumbnail_for_path.cache_info().currsize == 0
