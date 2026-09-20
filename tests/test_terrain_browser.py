"""The visual terrain browser's Qt wiring: swatches, grouping, filtering,
the toolbar button's gating, and the accept path that writes back into
terrain_combo.

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


def _dialog(current_id: int | None = _GRASS_1):
    conftest.ensure_qapp()
    return terrain_browser.TerrainBrowseDialog(current_id)


def _rows(tree):
    for i in range(tree.topLevelItemCount()):
        group = tree.topLevelItem(i)
        for c in range(group.childCount()):
            yield group, group.child(c)


def _terrain_window():
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("draw")
    QApplication.processEvents()
    return window


# -- the dialog --------------------------------------------------------------


def test_every_terrain_appears_under_exactly_one_group() -> None:
    dialog = _dialog()
    try:
        values = [row.data(0, _user_role()) for _group, row in _rows(dialog.tree)]
        assert sorted(values) == sorted(e.id for e in terrain_catalog.terrains())
    finally:
        dialog.close()


def _user_role():
    from PyQt5.QtCore import Qt

    return Qt.UserRole


def test_the_junk_terrains_are_hidden_until_opted_into() -> None:
    dialog = _dialog()
    try:
        hidden_ids = {e.id for e in terrain_catalog.terrains() if e.hidden}
        shown = {
            row.data(0, _user_role())
            for _group, row in _rows(dialog.tree)
            if not row.isHidden()
        }
        assert not (shown & hidden_ids)
        dialog.show_hidden_checkbox.setChecked(True)
        shown = {
            row.data(0, _user_role())
            for _group, row in _rows(dialog.tree)
            if not row.isHidden()
        }
        assert hidden_ids <= shown
    finally:
        dialog.close()


def test_it_opens_on_the_current_terrains_category_only() -> None:
    """Load-bearing for open cost, not just for convenience: with no
    default_group every group expands, and an all-expanded tree decodes all
    85 distinct textures at once (measured at ~2.8s cold)."""
    dialog = _dialog(_SNOW_SOFT)
    try:
        expanded = [
            dialog.tree.topLevelItem(i).text(0)
            for i in range(dialog.tree.topLevelItemCount())
            if dialog.tree.topLevelItem(i).isExpanded()
        ]
        assert len(expanded) == 1
        assert expanded[0].startswith("Snow & Ice")
        assert dialog.tree.currentItem().data(0, _user_role()) == _SNOW_SOFT
    finally:
        dialog.close()


def test_expanded_rows_carry_a_swatch_icon_with_no_install() -> None:
    """The no-install path must still show colour, not blanks -- that is the
    degradation contract color_for_terrain_id exists for."""
    dialog = _dialog(_GRASS_1)
    try:
        assert not asset_source.is_available(), "conftest should hide the install here"
        expanded_rows = [
            row for group, row in _rows(dialog.tree) if group.isExpanded()
        ]
        assert expanded_rows
        for row in expanded_rows:
            assert not row.icon(0).isNull(), row.text(0)
    finally:
        dialog.close()


def test_collapsed_rows_stay_undecoded_until_expanded() -> None:
    dialog = _dialog(_GRASS_1)
    try:
        collapsed = [
            (group, row) for group, row in _rows(dialog.tree) if not group.isExpanded()
        ]
        assert collapsed
        group, row = collapsed[0]
        assert row.icon(0).isNull()
        group.setExpanded(True)
        assert not row.icon(0).isNull()
    finally:
        dialog.close()


def test_filtering_reveals_rows_with_their_icons_populated() -> None:
    """itemExpanded fires on _apply_filter's own programmatic setExpanded,
    which is what covers a filter-revealed row -- verified here rather than
    assumed, since a silent failure would mean iconless rows exactly when
    the user is searching."""
    dialog = _dialog(_GRASS_1)
    try:
        dialog.filter_edit.setText("snow")
        revealed = [row for _group, row in _rows(dialog.tree) if not row.isHidden()]
        assert revealed
        assert all("snow" in row.text(0).lower() for row in revealed)
        assert all(not row.icon(0).isNull() for row in revealed)
    finally:
        dialog.close()


def test_the_raw_enum_name_is_searchable_too() -> None:
    """The display label is title-cased with spaces, so filtering on the
    underscored enum name would otherwise miss."""
    dialog = _dialog(_GRASS_1)
    try:
        dialog.filter_edit.setText("non_navigable")
        revealed = [row for _group, row in _rows(dialog.tree) if not row.isHidden()]
        assert revealed
    finally:
        dialog.close()


def test_each_swatch_shows_its_own_terrains_colour() -> None:
    """Measure, don't eyeball: a browser of 131 identical squares would sail
    through a "did it render" check. Asserted as "the swatch matches this
    terrain's own palette colour" rather than "every category differs" --
    the offscreen platform rounds a filled pixmap's channels by a few
    counts, which collapses two near-neighbour categories into one value
    and says nothing about the swatch being right."""
    from descape import terrain_palette

    dialog = _dialog(_GRASS_1)
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
        dialog.close()


# -- toolbar wiring ----------------------------------------------------------


def test_the_browse_button_tracks_the_terrain_combos_visibility() -> None:
    window = _terrain_window()
    try:
        for tool in ("draw", "elevation", "draw_rect", "set_level", "fill"):
            window._on_tool_selected(tool)
            assert (
                window.terrain_param_browse_action.isVisible()
                == window.terrain_param_combo_action.isVisible()
            ), tool
            assert window.terrain_browse_button.isEnabled() == window.terrain_combo.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_accepting_the_dialog_drives_the_terrain_combo() -> None:
    from PyQt5.QtWidgets import QDialog

    window = _terrain_window()
    try:
        window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(_GRASS_1))
        dialog = terrain_browser.TerrainBrowseDialog(_GRASS_1, parent=window)
        dialog.select(_SNOW_SOFT)
        dialog._accept_current()
        assert dialog.result() == QDialog.Accepted
        assert dialog.selected_id() == _SNOW_SOFT

        # The same write-back _browse_terrains does on accept.
        window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(dialog.selected_id()))
        assert window.terrain_combo.currentData() == _SNOW_SOFT
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_cancelled_dialog_leaves_the_combo_alone(monkeypatch) -> None:
    from PyQt5.QtWidgets import QDialog

    window = _terrain_window()
    try:
        window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(_GRASS_1))
        monkeypatch.setattr(
            terrain_browser.TerrainBrowseDialog, "exec_", lambda self: QDialog.Rejected
        )
        window._browse_terrains()
        assert window.terrain_combo.currentData() == _GRASS_1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_browse_handler_writes_the_picked_terrain_back(monkeypatch) -> None:
    from PyQt5.QtWidgets import QDialog

    window = _terrain_window()
    try:
        window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(_GRASS_1))

        def _fake_exec(dialog):
            dialog.select(_SNOW_SOFT)
            dialog._selected_value = _SNOW_SOFT
            return QDialog.Accepted

        monkeypatch.setattr(terrain_browser.TerrainBrowseDialog, "exec_", _fake_exec)
        window._browse_terrains()
        assert window.terrain_combo.currentData() == _SNOW_SOFT
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- cache registration ------------------------------------------------------


def test_changing_the_install_clears_the_thumbnail_cache() -> None:
    """The plan's own "this will bite" item: miss the clear list and the
    browser shows stale swatches after a GUI install change."""
    asset_source.get_terrain_thumbnail(_GRASS_1, 8)
    asset_source.set_install_path_override(None)
    assert asset_source._terrain_thumbnail_for_path.cache_info().currsize == 0
