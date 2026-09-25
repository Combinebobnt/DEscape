"""GH #41's trigger map overlay, driven through a real offscreen ViewerWindow:
panel selection -> trigger_geometry -> MapView scene items, the re-derive
paths (field edit, structural edit, undo), the lifecycle hooks, and pixel
captures at 1:1 in all three terrain styles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
GEOMETRY_TRIGGER = 3
CONDITION_AREA = ("condition", 1)
PATROL = ("effect", 1)
CREATE = ("effect", 2)
WHOLE_MAP = ("effect", 4)
OUTLINE_RGB = (255, 0, 255)


def _gen():
    return conftest.load_verify_module("gen_trigger_fixture")


def _window(tmp_path=None, monkeypatch=None, *, style: str = "Stepped", outline: str | None = None):
    if outline is not None:
        from descape import settings

        (tmp_path / "config.yaml").write_text(f"overlay_colors:\n  trigger_area_outline: '{outline}'\n")
        monkeypatch.setattr(settings, "_overlay_colors", None)
    window = conftest.shown_window(900, 700)
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None
    window.terrain_style_combo.setCurrentText(style)
    window.mode_combo.setCurrentText("Triggers")
    window.trigger_panel.select_trigger(GEOMETRY_TRIGGER)
    return window


def _shape_refs(view):
    return [(s.entry_ref, s.shape, s.coords) for s in view._trigger_shapes]


def _items_by_role(view):
    return {item.data(0): item for item in view.trigger_overlay_items()}


def _labels(view):
    return [i.text() for i in view.trigger_overlay_items() if hasattr(i, "text")]


# -- selection -> overlay ------------------------------------------------------


def test_selecting_the_trigger_draws_every_shape_unemphasised() -> None:
    gen = _gen()
    window = _window()
    try:
        view = window.map_view
        # Referenced units are pinned by the unit tests at the end of this file.
        assert [r for r in _shape_refs(view) if r[1] != "unit"] == [
            (CONDITION_AREA, "area", gen.GEOMETRY_CONDITION_AREA),
            (PATROL, "area", gen.GEOMETRY_PATROL_AREA),
            (PATROL, "location", gen.GEOMETRY_PATROL_LOCATION),
            (CREATE, "location", gen.GEOMETRY_CREATE_LOCATION),
            (WHOLE_MAP, "area", gen.GEOMETRY_WHOLE_MAP_AREA),
            (("effect", gen.TASK_OBJECT_EFFECT), "location", gen.HOUSE_TILE),
        ]
        assert view._trigger_emphasis is None
        assert view.trigger_overlay_items()
    finally:
        conftest.close_window(window)


def test_selecting_an_entry_emphasises_it_and_a_coordinate_free_trigger_draws_nothing() -> None:
    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*PATROL)
        assert view._trigger_emphasis == PATROL
        window.trigger_panel.select_trigger(0)
        assert view._trigger_shapes == [] and view.trigger_overlay_items() == []
    finally:
        conftest.close_window(window)


def test_the_panel_hook_fires_on_user_selection_not_on_populate() -> None:
    window = _window()
    try:
        panel = window.trigger_panel
        calls = []
        panel.on_focus_changed = lambda: calls.append(1)
        panel.select_trigger(1)
        assert calls, "a trigger change did not reach the hook"
        calls.clear()
        panel.select_entry("effect", 0)
        assert calls, "an entry change did not reach the hook"
        calls.clear()
        panel._populating = True
        try:
            panel._sync_entry_form()
        finally:
            panel._populating = False
        assert calls == [], "a programmatic populate fired the hook"
    finally:
        conftest.close_window(window)


def test_a_multi_selection_clears_the_overlay() -> None:
    window = _window()
    try:
        window.trigger_panel.select_triggers([GEOMETRY_TRIGGER, 0])
        assert window.map_view._trigger_shapes == []
        window.trigger_panel.select_trigger(GEOMETRY_TRIGGER)
        assert window.map_view._trigger_shapes
    finally:
        conftest.close_window(window)


def test_the_run_label_reads_the_rulers_own_number() -> None:
    from descape import ruler

    gen = _gen()
    window = _window()
    try:
        window.trigger_panel.select_entry(*PATROL)
        expected = ruler.format_measurement(ruler.measure((32, 43), gen.GEOMETRY_PATROL_LOCATION))
        assert _labels(window.map_view) == [expected]
    finally:
        conftest.close_window(window)


def test_an_unparseable_trigger_section_clears_the_overlay(monkeypatch) -> None:
    import descape.viewer as viewer_module

    window = _window()
    try:
        assert window.map_view._trigger_shapes
        # A scoped context: a bare undo() would also revert the autouse settings isolation.
        with monkeypatch.context() as patch:
            patch.setattr(viewer_module, "parse_triggers", lambda loaded: None)
            window._refresh_trigger_overlay()
        assert window.map_view._trigger_shapes == [] and window.map_view.trigger_overlay_items() == []
    finally:
        conftest.close_window(window)


# -- re-derive after edits -------------------------------------------------------


def _area_x2_spec(window):
    from descape import library_compat, trigger_fields

    vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)
    definition = vocabulary.effects[19]  # patrol
    specs = trigger_fields.field_specs(definition, vocabulary.effect_presentation, "effect_type")
    return next(s for s in specs if s.attribute == "area_x2")


def _patrol_area(view):
    return next(s.coords for s in view._trigger_shapes if s.entry_ref == PATROL and s.shape == "area")


def _outline_path_rect(view):
    return _items_by_role(view)["outline_strong"].path().boundingRect()


def _outline_path(view):
    from PyQt5.QtGui import QPainterPath

    return QPainterPath(_items_by_role(view)["outline_strong"].path())


def test_a_field_edit_moves_the_outline_and_undo_moves_it_back_even_in_view_mode() -> None:
    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*PATROL)
        before_rect = _outline_path_rect(view)
        window.set_entry_fields(GEOMETRY_TRIGGER, [PATROL], _area_x2_spec(window), 44)
        assert _patrol_area(view)[2] == 44
        assert _outline_path_rect(view) != before_rect, "the outline did not move"
        window.mode_combo.setCurrentText("View")
        window.undo()
        assert _patrol_area(view)[2] == 34
        assert _outline_path_rect(view) == before_rect
    finally:
        conftest.close_window(window)


def test_a_structural_delete_re_derives_the_entry_indices() -> None:
    window = _window()
    try:
        view = window.map_view
        panel = window.trigger_panel
        panel.select_entry(*PATROL)
        patrol_rect = _outline_path_rect(view)
        # As in the GUI: the effect above is selected to delete it.
        panel.select_entry("effect", 0)
        before_delete = _overlay_snapshot(view)
        window.entry_structural_edit("delete", GEOMETRY_TRIGGER, "effect", [("effect", 0)], -1)
        # The change_variable effect is gone: the patrol is effect 0 now.
        assert any(s.entry_ref == ("effect", 0) and s.shape == "area" for s in view._trigger_shapes)
        # The whole-map area moved down to effect 3 with it.
        assert any(s.entry_ref == ("effect", 3) and s.coords == _gen().GEOMETRY_WHOLE_MAP_AREA for s in view._trigger_shapes)
        assert not any(s.entry_ref == ("effect", 4) and s.shape == "area" for s in view._trigger_shapes)
        # GH #41: the delete lands on the patrol, and the emphasised ring is the patrol's.
        assert panel.current_entry_ref() == ("effect", 0)
        assert _outline_path_rect(view) == patrol_rect
        assert _overlay_snapshot(view) != before_delete
        window.undo()
        assert any(s.entry_ref == WHOLE_MAP for s in view._trigger_shapes)
        # Back to the pre-delete state: the restored effect above selected, the same items drawn.
        assert panel.current_entry_ref() == ("effect", 0)
        assert _overlay_snapshot(view) == before_delete
    finally:
        conftest.close_window(window)


def _overlay_snapshot(view) -> dict:
    """Each overlay item's role and scene bounds: what the user sees drawn."""
    return {item.data(0): item.boundingRect() for item in view.trigger_overlay_items()}


# -- lifecycle ---------------------------------------------------------------------


def test_the_overlay_survives_a_mode_change_and_a_restyle() -> None:
    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*PATROL)
        shapes = list(view._trigger_shapes)
        window.mode_combo.setCurrentText("View")
        assert view._trigger_shapes == shapes and view.trigger_overlay_items()
        for style in ("Sloped", "Flat", "Stepped"):
            window.terrain_style_combo.setCurrentText(style)
            assert view._trigger_shapes == shapes, style
            assert view._trigger_emphasis == PATROL, style
            assert all(item.scene() is view.scene() for item in view.trigger_overlay_items()), style
    finally:
        conftest.close_window(window)


def test_close_and_reload_leave_no_overlay() -> None:
    window = _window()
    try:
        view = window.map_view
        window.close_scenario()
        assert view._trigger_shapes == [] and view.trigger_overlay_items() == []
        window.load_scenario(FIXTURE_PATH)
        assert view._trigger_shapes == [] and view.trigger_overlay_items() == []
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_elevation_edit_moves_the_outline(style: str) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    gen = _gen()
    window = _window(style=style)
    try:
        view = window.map_view
        assert view._terrain_style == style.lower(), "the restyle did not reach the view"
        window.trigger_panel.select_entry(*CONDITION_AREA)
        before = _outline_path(view)
        x1, y1, _x2, _y2 = gen.GEOMETRY_CONDITION_AREA
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(x1, y1, Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        assert _outline_path(view) != before, "the outline stayed at the pre-edit ground height"
    finally:
        conftest.close_window(window)


def test_the_view_toggle_hides_without_dropping_and_persists(monkeypatch) -> None:
    from descape import settings

    window = _window()
    try:
        view = window.map_view
        assert window.trigger_overlay_action.isChecked()
        window.trigger_overlay_action.setChecked(False)
        assert settings.get_trigger_overlay() is False
        assert view.trigger_overlay_items() and not any(i.isVisible() for i in view.trigger_overlay_items())
        window.trigger_panel.select_entry(*PATROL)
        assert not any(i.isVisible() for i in view.trigger_overlay_items()), "a rebuild re-showed hidden items"
        window.trigger_overlay_action.setChecked(True)
        assert all(i.isVisible() for i in view.trigger_overlay_items())
        window.trigger_overlay_action.setChecked(False)
    finally:
        conftest.close_window(window)

    # GH #41's restart: drop the memo so the fresh window has to read config.yaml back.
    monkeypatch.setattr(settings, "_trigger_overlay", None)
    fresh = _window()
    try:
        assert not fresh.trigger_overlay_action.isChecked()
        items = fresh.map_view.trigger_overlay_items()
        assert items and not any(i.isVisible() for i in items)
    finally:
        conftest.close_window(fresh)


def test_the_z_band_sits_between_the_selection_and_the_region_and_above_the_ruler_label() -> None:
    from descape.map_view import MapView

    assert MapView.UNIT_SELECT_OUTLINE_Z < MapView.TRIGGER_AREA_Z < MapView.TRIGGER_MARK_Z < MapView.REGION_SELECT_Z
    assert MapView.RULER_LABEL_Z < MapView.TRIGGER_LABEL_Z < MapView.ANALYSIS_MARKER_Z < MapView.MIRROR_OVERLAY_Z
    window = _window()
    try:
        window.trigger_panel.select_entry(*PATROL)
        z = {role: item.zValue() for role, item in _items_by_role(window.map_view).items()}
        assert z["outline_strong"] == z["fill_strong"] == MapView.TRIGGER_AREA_Z
        assert z["marks_strong"] == MapView.TRIGGER_MARK_Z
        assert z["label"] == MapView.TRIGGER_LABEL_Z
    finally:
        conftest.close_window(window)


def test_a_whole_map_outline_is_o_perimeter_not_o_area() -> None:
    """The assertion that stops a refactor reintroducing one polygon per tile:
    a 120x120 area is 14400 tiles but only 480 border tiles. Doubling the side
    must roughly double the element count (O(perimeter)), not quadruple it."""
    for style in ("Stepped", "Sloped", "Flat"):
        window = _window(style=style)
        try:
            view = window.map_view
            window.trigger_panel.select_entry(*WHOLE_MAP)
            whole = _items_by_role(view)["outline_strong"].path().elementCount()
            half = view._rect_ring_path((0, 0, 60, 60)).elementCount()
            assert whole == view._rect_ring_path((0, 0, 120, 120)).elementCount(), style
            assert 1.8 <= whole / half <= 2.2, (style, whole, half)
            assert whole < 4 * 120 * 120, (style, whole)
        finally:
            conftest.close_window(window)


# -- pixels ----------------------------------------------------------------------


def _colour_hits(array, rgb, tolerance):
    return (np.abs(array.astype(np.int16) - np.array(rgb, dtype=np.int16)) <= tolerance).all(axis=2)


def _area_capture(view, area):
    from PyQt5.QtCore import QRectF
    from PyQt5.QtWidgets import QApplication

    x1, y1, x2, y2 = area
    bounds = None
    for tx, ty in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        rect = view._tile_polygon(tx, ty).boundingRect()
        bounds = rect if bounds is None else bounds.united(rect)
    rect = QRectF(bounds.adjusted(-8, -8, 8, 8).toAlignedRect())
    QApplication.processEvents()
    return conftest.scene_rect_to_array(view.scene(), rect)


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_the_area_outline_paints_in_every_style_and_emphasis_is_stronger(tmp_path, monkeypatch, style) -> None:
    gen = _gen()
    window = _window(tmp_path, monkeypatch, style=style, outline="#ff00ff")
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*CONDITION_AREA)
        strong = _area_capture(view, gen.GEOMETRY_CONDITION_AREA)
        dimmed = _area_capture(view, gen.GEOMETRY_PATROL_AREA)
        strong_hits = int(_colour_hits(strong, OUTLINE_RGB, 12).sum())
        dim_hits = int(_colour_hits(dimmed, OUTLINE_RGB, 12).sum())
        assert strong_hits >= 40, (style, strong_hits)
        assert dim_hits == 0, (style, dim_hits)
        # Dimmed, not gone: the same hue at reduced alpha still marks the area.
        assert _colour_hits(dimmed, OUTLINE_RGB, 150).sum() > 0, style
    finally:
        conftest.close_window(window)


# -- referenced units (trigger unit references, slice C) ---------------------------


def _unit_shapes(view):
    return [(s.entry_ref, s.fields[0], s.coords) for s in view._trigger_shapes if s.shape == "unit"]


def test_referenced_units_are_outlined_and_the_task_runs_to_its_location_object() -> None:
    gen = _gen()
    window = _window()
    try:
        view = window.map_view
        house = (40, 62, 41, 63)
        assert (("condition", gen.DESTROY_CONDITION), "unit_object", house) in _unit_shapes(view)
        assert (("effect", gen.TASK_OBJECT_EFFECT), "location_object_reference", house) in _unit_shapes(view)
        roles = _items_by_role(view)
        assert "units_strong" in roles and "fill_units_strong" not in roles
        window.trigger_panel.select_entry("effect", gen.TASK_OBJECT_EFFECT)
        roles = _items_by_role(view)
        strong, dim = roles["marks_strong"].path(), roles["marks_dim"].path()
        centre = view._location_mark(gen.HOUSE_TILE).boundingRect().center()
        # The stored location is an authoring copy of the house's tile: dimmed, never strong.
        assert dim.contains(centre) and not strong.contains(centre)
        assert _labels(view), "the task's run, archers' centroid to the house, is labelled"
    finally:
        conftest.close_window(window)


def test_moving_a_referenced_unit_moves_its_outline() -> None:
    gen = _gen()
    window = _window()
    try:
        view = window.map_view
        before = _items_by_role(view)["units_strong"].path().boundingRect()
        unit = next(u for u in window.scenario.unit_manager.units[1] if u.reference_id == gen.REF_HOUSE)
        unit.x, unit.y = 81.0, 21.0
        window._after_unit_mutation()
        moved = [c for e, f, c in _unit_shapes(view) if f == "unit_object"]
        assert moved == [(80, 20, 81, 21)]
        after = _items_by_role(view)["units_strong"].path().boundingRect()
        assert after != before
    finally:
        conftest.close_window(window)


def test_the_unit_outline_is_on_its_own_colour_row(tmp_path, monkeypatch) -> None:
    from descape import settings

    (tmp_path / "config.yaml").write_text("overlay_colors:\n  trigger_unit_ref: '#ff00ff'\n")
    monkeypatch.setattr(settings, "_overlay_colors", None)
    window = _window()
    try:
        pen = _items_by_role(window.map_view)["units_strong"].pen()
        assert pen.color().name() == "#ff00ff"
    finally:
        conftest.close_window(window)


def test_a_colour_change_reinks_a_live_outline() -> None:
    """GH #41: Settings > Appearance recolours a drawn overlay straight away,
    through the same call its colour swatch makes."""
    from descape import settings

    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*PATROL)
        assert _items_by_role(view)["outline_strong"].pen().color().name() != "#ff00ff"
        settings.set_overlay_color("trigger_area_outline", "#ff00ff")
        view.apply_overlay_colors()
        # Re-fetched: a colour change rebuilds the items rather than re-inking them.
        assert _items_by_role(view)["outline_strong"].pen().color().name() == "#ff00ff"
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_the_unit_outline_paints_on_the_house_footprint_in_every_style(tmp_path, monkeypatch, style) -> None:
    from descape import settings

    (tmp_path / "config.yaml").write_text("overlay_colors:\n  trigger_unit_ref: '#ff00ff'\n")
    monkeypatch.setattr(settings, "_overlay_colors", None)
    window = _window(style=style)
    try:
        view = window.map_view
        # Only the house's own entry, so no area outline shares the frame.
        window.trigger_panel.select_entry("condition", _gen().DESTROY_CONDITION)
        array = _area_capture(view, (40, 62, 41, 63))
        assert _colour_hits(array, OUTLINE_RGB, 40).sum() > 20
        # Countable: the ring's extent is exactly the footprint tiles' own polygons.
        footprint = None
        for tx, ty in ((40, 62), (41, 62), (41, 63), (40, 63)):
            rect = view._tile_polygon(tx, ty).boundingRect()
            footprint = rect if footprint is None else footprint.united(rect)
        ring = _items_by_role(view)["units_strong"].path().boundingRect()
        for got, want in zip(ring.getCoords(), footprint.getCoords(), strict=True):
            assert abs(got - want) < 0.5, (style, ring, footprint)
    finally:
        conftest.close_window(window)
